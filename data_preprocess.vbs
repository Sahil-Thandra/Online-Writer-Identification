Set fso = CreateObject("Scripting.FileSystemObject")

sFolder1 = "D:\Handwriting-Dataset3\Data1\Update_chinese1\"
sFolder2 = "D:\Handwriting-Dataset3\Data1\Update_samechinese\"
'sFolder1 = "D:\Handwriting-Dataset3\Data1\Update_samechinese\"

Set folder = fso.GetFolder(sFolder1)
Set collFiles = folder.Files

Set readme = fso.CreateTextFile("D:\Handwriting-Dataset3\Data1\readme.txt" )
Set trainFile = fso.CreateTextFile("D:\Handwriting-Dataset3\Data1\train.dta" )

itr = 1
for each fileIdx in collFiles
	sFile1Path = sFolder1 & fileIdx.Name
	sFile2Path = sFolder2 & fileIdx.Name
	
	fileExist = fso.FileExists(sFile2Path)
	if not fileExist then
	wscript.echo "File doesnot exists" & sFile2Path
	else
	
	Set objTextFile = fso.OpenTextFile(sFile1Path, 1)
	counter = 0
	
	Do Until (objTextFile.AtEndOfStream)
	counter = counter + 1
    strLine = objTextFile.Readline
    strLine = Trim(strLine)
    If Len(strLine) > 0 Then
        trainFile.WriteLine strLine
    End If
	 
	Loop
	readme.writeline itr & "," & fileIdx.Name
	objTextFile.Close
	
	Set objTextFile = fso.OpenTextFile(sFile2Path, 1)
	
	counter = 0
	Do Until (objTextFile.AtEndOfStream)
	counter = counter+1
    strLine = objTextFile.Readline
    strLine = Trim(strLine)
    If Len(strLine) > 0 Then
        trainFile.WriteLine strLine
    End If
	Loop
	objTextFile.Close
	
	
	itr = itr+1
	if itr = 10 then
		
	end if
	end if
next

sFolder1 = "D:\Handwriting-Dataset3\Data1\Update_chinese2\10writers\"

Set folder = fso.GetFolder(sFolder1)
Set collFiles = folder.Files

Set readme = fso.CreateTextFile("D:\Handwriting-Dataset3\Data1\readme.txt" )
Set testFile = fso.CreateTextFile("D:\Handwriting-Dataset3\Data1\test10writers.dta" )

itr = 1
for each fileIdx in collFiles
	sFile1Path = sFolder1 & fileIdx.Name
	Set objTextFile = fso.OpenTextFile(sFile1Path, 1)
	
	counter = 0
	Do Until (objTextFile.AtEndOfStream)
	counter = counter + 1
    strLine = objTextFile.Readline
    strLine = Trim(strLine)
    If Len(strLine) > 0 Then
        testFile.WriteLine strLine
    End If
	Loop
	objTextFile.Close
	
	readme.writeline itr
	itr = itr+1

next
