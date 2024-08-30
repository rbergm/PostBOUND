SELECT COUNT(*)
FROM comments AS c, badges AS b, users AS u
WHERE c.UserId = u.Id
  AND b.UserId = u.Id
  AND c.CreationDate >= CAST('2010-08-12 20:27:30' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-12 12:49:19' AS timestamp)
  AND u.Views >= 0
  AND u.DownVotes >= 0
  AND u.DownVotes <= 2;
