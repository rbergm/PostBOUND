SELECT COUNT(*)
FROM comments AS c, badges AS b, users AS u
WHERE u.Id = c.UserId
  AND c.UserId = b.UserId
  AND c.Score = 0
  AND b.Date >= CAST('2010-07-19 20:54:06' AS timestamp)
  AND u.DownVotes >= 0
  AND u.UpVotes >= 0
  AND u.UpVotes <= 17
  AND u.CreationDate >= CAST('2010-08-06 07:03:05' AS timestamp)
  AND u.CreationDate <= CAST('2014-09-08 04:18:44' AS timestamp);
